"use client";

import { useState, useEffect } from "react";
import { Music, Video, Gamepad2, Radio, Heart, ArrowLeft, Play, Pause, SkipBack, SkipForward, Volume2, Book } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

const categories = [
  { id: "music", name: "Music", icon: Music, color: "bg-purple-500" },
  { id: "video", name: "Videos", icon: Video, color: "bg-accent" },
  { id: "games", name: "Games", icon: Gamepad2, color: "bg-success" },
  { id: "radio", name: "Radio", icon: Radio, color: "bg-secondary" },
  { id: "stories", name: "Stories", icon: Book, color: "bg-indigo-500" },
];

const musicTracks = [
  { id: 1, title: "Relaxing Piano", artist: "Calm Sounds", duration: "4:32", mood: "relaxing" },
  { id: 2, title: "Morning Jazz", artist: "Smooth Jazz", duration: "5:17", mood: "upbeat" },
  { id: 3, title: "Nature Sounds", artist: "Ambient", duration: "10:00", mood: "relaxing" },
  { id: 4, title: "Classical Favorites", artist: "Orchestra", duration: "6:45", mood: "focus" },
  { id: 5, title: "Feel Good Hits", artist: "Various Artists", duration: "3:52", mood: "upbeat" },
];

const games = [
  { id: 1, name: "Memory Match", description: "Match the pairs", difficulty: "Easy", icon: "🧩" },
  { id: 2, name: "Word Puzzle", description: "Find hidden words", difficulty: "Medium", icon: "📝" },
  { id: 3, name: "Sudoku", description: "Number challenge", difficulty: "Hard", icon: "🔢" },
  { id: 4, name: "Trivia Quiz", description: "Test your knowledge", difficulty: "Easy", icon: "❓" },
];

// ==========================================
// 1. Memory Match Mini-Game
// ==========================================
function MemoryMatchGame({ onBack }: { onBack: () => void }) {
  const emojis = ['🤖', '🍎', '🐶', '🚗', '🌈', '⭐'];
  const [cards, setCards] = useState<any[]>([]);
  const [flipped, setFlipped] = useState<number[]>([]);
  const [matches, setMatches] = useState<number[]>([]);
  const [moves, setMoves] = useState(0);

  const initGame = () => {
    const shuffled = [...emojis, ...emojis]
      .map((emoji, index) => ({ id: index, emoji }))
      .sort(() => Math.random() - 0.5);
    setCards(shuffled);
    setFlipped([]);
    setMatches([]);
    setMoves(0);
  };

  useEffect(() => {
    initGame();
  }, []);

  const handleCardClick = (id: number) => {
    if (flipped.length === 2 || flipped.includes(id) || matches.includes(id)) return;

    const newFlipped = [...flipped, id];
    setFlipped(newFlipped);

    if (newFlipped.length === 2) {
      setMoves(prev => prev + 1);
      const [firstId, secondId] = newFlipped;
      if (cards[firstId].emoji === cards[secondId].emoji) {
        setMatches(prev => [...prev, firstId, secondId]);
        setFlipped([]);
      } else {
        setTimeout(() => setFlipped([]), 1000);
      }
    }
  };

  const won = matches.length === 12;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-3xl p-6 border-2 border-success/30 dark:border-success/50 shadow-soft max-w-md mx-auto text-center space-y-6 animate-scale-up">
      <div className="flex justify-between items-center">
        <button onClick={onBack} className="px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-xl font-semibold transition-colors">
          ← Back
        </button>
        <span className="text-xl font-bold text-success">Memory Match</span>
        <button onClick={initGame} className="px-4 py-2 bg-success text-white rounded-xl font-semibold hover:bg-success-600 transition-colors">
          Reset
        </button>
      </div>

      <div className="flex justify-around text-text-secondary dark:text-gray-300 font-semibold text-lg">
        <span>Moves: {moves}</span>
        <span>Matches: {matches.length / 2} / 6</span>
      </div>

      {won ? (
        <div className="py-8 space-y-4">
          <div className="text-6xl animate-bounce">🏆</div>
          <h3 className="text-2xl font-black text-success">You Won!</h3>
          <p className="text-text-secondary dark:text-gray-300">Completed in {moves} moves.</p>
          <button onClick={initGame} className="w-full py-3 bg-success hover:bg-success-600 text-white rounded-2xl font-bold transition-all shadow-md">
            Play Again
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-4 gap-3">
          {cards.map((card, idx) => {
            const isCardFlipped = flipped.includes(idx) || matches.includes(idx);
            return (
              <button
                key={card.id}
                onClick={() => handleCardClick(idx)}
                className={cn(
                  "aspect-square rounded-2xl flex items-center justify-center text-3xl font-bold transition-all duration-300 transform shadow-sm",
                  isCardFlipped
                    ? "bg-success text-white rotate-0"
                    : "bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 rotate-180"
                )}
              >
                <span className={cn(isCardFlipped ? "opacity-100 scale-100" : "opacity-0 scale-50", "transition-all duration-300")}>
                  {card.emoji}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ==========================================
// 2. Word Puzzle Mini-Game
// ==========================================
function WordPuzzleGame({ onBack }: { onBack: () => void }) {
  const puzzles = [
    { word: "ROBOT", hint: "A physical mechanical agent that performs tasks" },
    { word: "HEALTH", hint: "The state of being free from illness or injury" },
    { word: "HAPPY", hint: "Feeling or showing pleasure or contentment" },
    { word: "MUSIC", hint: "Vocal or instrumental sounds combined to produce beauty of form" },
  ];
  
  const [index, setIndex] = useState(0);
  const [scrambled, setScrambled] = useState("");
  const [input, setInput] = useState("");
  const [message, setMessage] = useState("");
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  
  const current = puzzles[index];

  const initPuzzle = () => {
    const word = puzzles[index].word;
    const shuffled = word.split('').sort(() => Math.random() - 0.5).join('');
    setScrambled(shuffled === word ? shuffled.split('').reverse().join('') : scrambled);
    setInput("");
    setMessage("");
    setIsCorrect(null);
  };

  useEffect(() => {
    initPuzzle();
  }, [index]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.toUpperCase().trim() === current.word) {
      setIsCorrect(true);
      setMessage("🎉 Correct! Fantastic job!");
    } else {
      setIsCorrect(false);
      setMessage("❌ Try again, you can do it!");
    }
  };

  const next = () => {
    setIndex((index + 1) % puzzles.length);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-3xl p-6 border-2 border-secondary/30 dark:border-secondary/50 shadow-soft max-w-md mx-auto text-center space-y-6 animate-scale-up">
      <div className="flex justify-between items-center">
        <button onClick={onBack} className="px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-xl font-semibold transition-colors">
          ← Back
        </button>
        <span className="text-xl font-bold text-secondary">Word Puzzle</span>
        <button onClick={initPuzzle} className="px-4 py-2 bg-secondary text-white rounded-xl font-semibold hover:bg-secondary-600 transition-colors">
          Reset
        </button>
      </div>

      <div className="space-y-2">
        <p className="text-sm text-text-muted dark:text-gray-400 font-semibold tracking-widest uppercase">Scrambled Word</p>
        <div className="text-4xl font-extrabold tracking-wider text-secondary animate-pulse">{scrambled}</div>
      </div>

      <div className="bg-gray-50 dark:bg-gray-900/50 rounded-2xl p-4 border border-gray-100 dark:border-gray-800 text-left">
        <p className="text-sm text-text-muted dark:text-gray-400 font-bold uppercase mb-1">Clue</p>
        <p className="text-base text-text-primary dark:text-white">{current.hint}</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Unscramble the word..."
          className="w-full px-5 py-4 border-2 border-gray-200 dark:border-gray-700 rounded-2xl text-center text-xl font-bold focus:border-secondary focus:ring-4 focus:ring-secondary/20 outline-none transition-all dark:bg-gray-900 text-text-primary dark:text-white"
        />
        <button type="submit" className="w-full py-4 bg-secondary hover:bg-secondary-600 text-white rounded-2xl text-lg font-bold transition-all shadow-md">
          Check Answer
        </button>
      </form>

      {message && (
        <div className="space-y-4">
          <p className={cn("text-lg font-bold", isCorrect ? "text-green-500" : "text-red-500")}>
            {message}
          </p>
          {isCorrect && (
            <button onClick={next} className="px-6 py-3 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-text-primary dark:text-white rounded-xl font-bold transition-colors">
              Next Word →
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ==========================================
// 3. Mini Sudoku Mini-Game
// ==========================================
function SudokuGame({ onBack }: { onBack: () => void }) {
  const initialGrid = [
    [1, 0, 3, 0],
    [0, 4, 0, 2],
    [2, 0, 4, 0],
    [0, 3, 0, 1]
  ];
  
  const solution = [
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1]
  ];

  const [grid, setGrid] = useState<number[][]>([]);
  const [win, setWin] = useState(false);

  useEffect(() => {
    setGrid(initialGrid.map(row => [...row]));
    setWin(false);
  }, []);

  const handleChange = (row: number, col: number, val: string) => {
    if (initialGrid[row][col] !== 0) return;
    
    const num = parseInt(val);
    const newGrid = grid.map((r, rIdx) => r.map((c, cIdx) => {
      if (rIdx === row && cIdx === col) {
        return (isNaN(num) || num < 1 || num > 4) ? 0 : num;
      }
      return c;
    }));
    
    setGrid(newGrid);

    // check win condition
    let isComplete = true;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        if (newGrid[r][c] !== solution[r][c]) {
          isComplete = false;
        }
      }
    }
    if (isComplete) setWin(true);
  };

  const reset = () => {
    setGrid(initialGrid.map(row => [...row]));
    setWin(false);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-3xl p-6 border-2 border-indigo-500/30 dark:border-indigo-500/50 shadow-soft max-w-md mx-auto text-center space-y-6 animate-scale-up">
      <div className="flex justify-between items-center">
        <button onClick={onBack} className="px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-xl font-semibold transition-colors">
          ← Back
        </button>
        <span className="text-xl font-bold text-indigo-500">Mini Sudoku (4x4)</span>
        <button onClick={reset} className="px-4 py-2 bg-indigo-500 text-white rounded-xl font-semibold hover:bg-indigo-600 transition-colors">
          Reset
        </button>
      </div>

      <p className="text-sm text-text-muted dark:text-gray-400">Fill in cells with numbers 1 to 4.</p>

      {win ? (
        <div className="py-8 space-y-4">
          <div className="text-6xl animate-bounce">🎉</div>
          <h3 className="text-2xl font-black text-indigo-500">Completed!</h3>
          <p className="text-text-secondary dark:text-gray-300">You solved the Sudoku grid perfectly!</p>
          <button onClick={reset} className="w-full py-3 bg-indigo-500 hover:bg-indigo-600 text-white rounded-2xl font-bold transition-all shadow-md">
            Play Again
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-4 gap-2 w-48 mx-auto border-4 border-indigo-500 bg-indigo-500 rounded-xl overflow-hidden p-1 shadow-md">
          {grid.map((row, rIdx) => 
            row.map((val, cIdx) => {
              const isClue = initialGrid[rIdx][cIdx] !== 0;
              return (
                <input
                  key={`${rIdx}-${cIdx}`}
                  type="text"
                  maxLength={1}
                  disabled={isClue}
                  value={val === 0 ? "" : val}
                  onChange={(e) => handleChange(rIdx, cIdx, e.target.value)}
                  className={cn(
                    "w-10 h-10 text-center text-xl font-black rounded outline-none transition-all",
                    isClue 
                      ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/50 dark:text-indigo-400 cursor-not-allowed" 
                      : "bg-white dark:bg-gray-900 text-text-primary dark:text-white focus:bg-indigo-50"
                  )}
                />
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

// ==========================================
// 4. Trivia Quiz Mini-Game
// ==========================================
function TriviaQuizGame({ onBack }: { onBack: () => void }) {
  const questions = [
    { question: "Which planet is known as the Red Planet?", options: ["Earth", "Mars", "Venus", "Jupiter"], answer: "Mars" },
    { question: "What is the primary color of NovaBot's theme?", options: ["Blue", "Purple", "Green", "Orange"], answer: "Purple" },
    { question: "How many legs does an insect have?", options: ["Four", "Six", "Eight", "Ten"], answer: "Six" },
  ];

  const [index, setIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [isAnswered, setIsAnswered] = useState(false);
  const [score, setScore] = useState(0);

  const handleOptionClick = (option: string) => {
    if (isAnswered) return;
    setSelectedOption(option);
    setIsAnswered(true);
    if (option === questions[index].answer) {
      setScore(prev => prev + 1);
    }
  };

  const next = () => {
    setIndex(index + 1);
    setSelectedOption(null);
    setIsAnswered(false);
  };

  const reset = () => {
    setIndex(0);
    setSelectedOption(null);
    setIsAnswered(false);
    setScore(0);
  };

  const current = questions[index];
  const finished = index >= questions.length;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-3xl p-6 border-2 border-primary/30 dark:border-primary/50 shadow-soft max-w-md mx-auto text-center space-y-6 animate-scale-up">
      <div className="flex justify-between items-center">
        <button onClick={onBack} className="px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-xl font-semibold transition-colors">
          ← Back
        </button>
        <span className="text-xl font-bold text-primary">Trivia Quiz</span>
        <button onClick={reset} className="px-4 py-2 bg-primary text-white rounded-xl font-semibold hover:bg-primary-600 transition-colors">
          Reset
        </button>
      </div>

      {finished ? (
        <div className="py-8 space-y-4">
          <div className="text-6xl animate-bounce">🎓</div>
          <h3 className="text-2xl font-black text-primary">Quiz Complete!</h3>
          <p className="text-text-secondary dark:text-gray-300 text-lg">Your score: <span className="font-extrabold text-primary text-2xl">{score} / {questions.length}</span></p>
          <button onClick={reset} className="w-full py-3 bg-primary hover:bg-primary-600 text-white rounded-2xl font-bold transition-all shadow-md">
            Restart Quiz
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="text-left space-y-2">
            <span className="text-sm font-semibold text-primary uppercase">Question {index + 1} of {questions.length}</span>
            <h3 className="text-xl font-bold text-text-primary dark:text-white leading-snug">{current.question}</h3>
          </div>

          <div className="grid gap-3">
            {current.options.map((option) => {
              const isSelected = selectedOption === option;
              const isCorrect = option === current.answer;
              const btnStyle = isAnswered
                ? isCorrect
                  ? "bg-green-500 border-green-500 text-white"
                  : isSelected
                    ? "bg-red-500 border-red-500 text-white"
                    : "bg-gray-50 border-gray-200 dark:bg-gray-900 dark:border-gray-700 opacity-60 text-text-muted"
                : "bg-white hover:bg-gray-50 dark:bg-gray-900 dark:hover:bg-gray-700 border-gray-200 dark:border-gray-700 text-text-primary dark:text-white hover:border-primary";
              
              return (
                <button
                  key={option}
                  disabled={isAnswered}
                  onClick={() => handleOptionClick(option)}
                  className={cn(
                    "w-full p-4 border-2 rounded-2xl text-left text-lg font-semibold transition-all flex items-center justify-between",
                    btnStyle
                  )}
                >
                  <span>{option}</span>
                  {isAnswered && isCorrect && <span>✓</span>}
                  {isAnswered && isSelected && !isCorrect && <span>✗</span>}
                </button>
              );
            })}
          </div>

          {isAnswered && (
            <button onClick={next} className="w-full py-4 bg-primary hover:bg-primary-600 text-white rounded-2xl text-lg font-bold transition-all shadow-md">
              {index + 1 === questions.length ? "Finish Quiz" : "Next Question →"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default function EntertainmentPage() {
  const [activeCategory, setActiveCategory] = useState("music");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTrack, setCurrentTrack] = useState<typeof musicTracks[0] | null>(null);
  const [activeGameId, setActiveGameId] = useState<number | null>(null);

  const playTrack = (track: typeof musicTracks[0]) => {
    setCurrentTrack(track);
    setIsPlaying(true);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link
          href="/rover"
          className="rover-btn w-14 h-14 rounded-2xl bg-gray-100 dark:bg-gray-800 flex items-center justify-center hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          <ArrowLeft className="w-6 h-6 text-text-secondary dark:text-gray-400" />
        </Link>
        <div>
          <h1 className="text-3xl font-display font-bold text-text-primary dark:text-white">Entertainment</h1>
          <p className="text-text-muted dark:text-gray-400">Music, videos, games, and more</p>
        </div>
      </div>

      {/* Category Tabs */}
      <div className="flex gap-3 overflow-x-auto pb-2">
        {categories.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setActiveCategory(cat.id)}
            className={cn(
              "rover-btn flex items-center gap-3 px-6 py-4 rounded-2xl transition-all whitespace-nowrap",
              activeCategory === cat.id
                ? `${cat.color} text-white`
                : "bg-white dark:bg-gray-800 border-2 border-gray-100 dark:border-gray-700 text-text-secondary dark:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600"
            )}
          >
            <cat.icon className="w-6 h-6" />
            <span className="text-lg font-semibold">{cat.name}</span>
          </button>
        ))}
      </div>

      {/* Music Section */}
      {activeCategory === "music" && (
        <div className="space-y-4">
          {/* Quick Moods */}
          <div className="flex gap-4">
            {["😌 Relaxing", "🎉 Upbeat", "🎯 Focus"].map((mood) => (
              <button
                key={mood}
                className="flex-1 py-4 bg-white dark:bg-gray-800 border-2 border-gray-100 dark:border-gray-700 rounded-2xl text-lg font-medium text-text-primary dark:text-white hover:border-primary hover:text-primary transition-colors"
              >
                {mood}
              </button>
            ))}
          </div>

          {/* Track List */}
          <div className="bg-white dark:bg-gray-800 rounded-3xl shadow-soft border border-gray-100 dark:border-gray-700 overflow-hidden">
            <div className="p-6 border-b border-gray-100 dark:border-gray-700">
              <h2 className="text-xl font-semibold text-text-primary dark:text-white">Your Music</h2>
            </div>
            <div className="divide-y divide-gray-100 dark:divide-gray-700">
              {musicTracks.map((track) => (
                <button
                  key={track.id}
                  onClick={() => playTrack(track)}
                  className={cn(
                    "w-full p-4 flex items-center gap-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-left",
                    currentTrack?.id === track.id && "bg-primary-50 dark:bg-primary-900/30"
                  )}
                >
                  <div className={cn(
                    "w-14 h-14 rounded-2xl flex items-center justify-center",
                    currentTrack?.id === track.id && isPlaying ? "bg-primary" : "bg-purple-100 dark:bg-purple-900/50"
                  )}>
                    {currentTrack?.id === track.id && isPlaying ? (
                      <Pause className="w-6 h-6 text-white" />
                    ) : (
                      <Play className="w-6 h-6 text-purple-500" />
                    )}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-text-primary dark:text-white">{track.title}</h3>
                    <p className="text-text-muted dark:text-gray-400">{track.artist}</p>
                  </div>
                  <span className="text-text-muted dark:text-gray-400">{track.duration}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Games Section */}
      {activeCategory === "games" && (
        activeGameId ? (
          <div className="w-full">
            {activeGameId === 1 && <MemoryMatchGame onBack={() => setActiveGameId(null)} />}
            {activeGameId === 2 && <WordPuzzleGame onBack={() => setActiveGameId(null)} />}
            {activeGameId === 3 && <SudokuGame onBack={() => setActiveGameId(null)} />}
            {activeGameId === 4 && <TriviaQuizGame onBack={() => setActiveGameId(null)} />}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            {games.map((game) => (
              <button
                key={game.id}
                onClick={() => setActiveGameId(game.id)}
                className="rover-card bg-white dark:bg-gray-800 rounded-3xl p-6 text-left border-2 border-gray-100 dark:border-gray-700 hover:border-success hover:shadow-lg transition-all"
              >
                <div className="text-5xl mb-4">{game.icon}</div>
                <h3 className="text-xl font-bold text-text-primary dark:text-white mb-1">{game.name}</h3>
                <p className="text-text-muted dark:text-gray-400 mb-3">{game.description}</p>
                <span className={cn(
                  "px-3 py-1 rounded-full text-sm font-medium",
                  game.difficulty === "Easy" ? "bg-success-100 dark:bg-success-900/50 text-success" :
                  game.difficulty === "Medium" ? "bg-secondary-100 dark:bg-secondary-900/50 text-secondary" :
                  "bg-accent-100 dark:bg-accent-900/50 text-accent"
                )}>
                  {game.difficulty}
                </span>
              </button>
            ))}
          </div>
        )
      )}

      {/* Video Section */}
      {activeCategory === "video" && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {["Nature Documentaries", "Cooking Shows", "Comedy Clips", "Travel Videos"].map((category) => (
              <button
                key={category}
                className="rover-card bg-gradient-to-br from-gray-700 to-gray-900 rounded-3xl p-8 text-white text-left hover:scale-[1.02] transition-all"
              >
                <Video className="w-10 h-10 mb-4 opacity-60" />
                <h3 className="text-xl font-bold">{category}</h3>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Radio Section */}
      {activeCategory === "radio" && (
        <div className="space-y-4">
          {["Jazz FM", "Classical Radio", "News Talk", "Oldies Station"].map((station) => (
            <button
              key={station}
              className="w-full p-6 bg-white rounded-2xl border-2 border-gray-100 flex items-center gap-4 hover:border-secondary transition-colors"
            >
              <div className="w-14 h-14 bg-secondary-100 rounded-2xl flex items-center justify-center">
                <Radio className="w-6 h-6 text-secondary" />
              </div>
              <span className="text-xl font-semibold text-text-primary">{station}</span>
            </button>
          ))}
        </div>
      )}

      {/* Stories Section */}
      {activeCategory === "stories" && (
        <div className="space-y-4">
          {["Bedtime Stories", "Audiobooks", "Podcasts", "Meditation"].map((type) => (
            <button
              key={type}
              className="w-full p-6 bg-white rounded-2xl border-2 border-gray-100 flex items-center gap-4 hover:border-indigo-500 transition-colors"
            >
              <div className="w-14 h-14 bg-indigo-100 rounded-2xl flex items-center justify-center">
                <Book className="w-6 h-6 text-indigo-500" />
              </div>
              <span className="text-xl font-semibold text-text-primary">{type}</span>
            </button>
          ))}
        </div>
      )}

      {/* Now Playing Bar */}
      {currentTrack && (
        <div className="fixed bottom-24 left-0 right-0 px-4">
          <div className="max-w-4xl mx-auto bg-gradient-to-r from-purple-600 to-purple-700 rounded-2xl p-4 shadow-elevated flex items-center gap-4">
            <div className="w-14 h-14 bg-white/20 rounded-xl flex items-center justify-center">
              <Music className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h3 className="text-white font-semibold">{currentTrack.title}</h3>
              <p className="text-white/60 text-sm">{currentTrack.artist}</p>
            </div>
            <div className="flex items-center gap-2">
              <button className="p-3 hover:bg-white/10 rounded-xl transition-colors">
                <SkipBack className="w-6 h-6 text-white" />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-4 bg-white rounded-xl hover:bg-gray-100 transition-colors"
              >
                {isPlaying ? (
                  <Pause className="w-6 h-6 text-purple-600" />
                ) : (
                  <Play className="w-6 h-6 text-purple-600" />
                )}
              </button>
              <button className="p-3 hover:bg-white/10 rounded-xl transition-colors">
                <SkipForward className="w-6 h-6 text-white" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
